

// API service for ZeeVice hybrid search backend
export interface HybridSearchResponse {
  response: string;                    // AI markdown
  search_results?: ProductCard[];      // list of product objects
  mentioned_product_ids?: string[];    // IDs mentioned in response
}
// export interface HybridSearchResponse {
//   query: string;
//   filters: {
//     category?: string;
//     brand?: string[];
//     store?: string[];
//     price_min?: number;
//     price_max?: number;
//   };
//   sim_flags: {
//     display_size?: boolean;
//     display_type?: boolean;
//     ram?: boolean;
//     storage?: boolean;
//     main_camera?: boolean;
//     front_camera?: boolean;
//     battery?: boolean;
//     cores?: boolean;
//     network?: boolean;
//     os?: boolean;
//     sim?: boolean;
//   };
//   ranked: Array<[string, number]>; // [(id, score)]
//   cards: ProductCard[];
// }

export interface ProductCard {
  id: string;
  title: string;
  brand: string;
  price: number;
  store: string;
  url?: string;
  display_size?: string;
  display_type?: string;
  ram?: string;
  storage?: string;
  battery?: string;
  cores?: string;
  os?: string;
  sim?: string;
  network?: string;
  main_camera?: string;
  front_camera?: string;
}

// export interface ChatResponse {
//   response: string; // always present
//   search_results?: HybridSearchResponse; // only for product_search
// }

export interface ChatResponse {
  response: string;
  search_results?: ProductCard[];       // actual product array
  mentioned_product_ids?: string[];     // optional array of IDs
}



class ZeeViceAPI {
  private baseURL: string;

  constructor() {
    // Update this to your Flask backend URL
    this.baseURL = "http://localhost:5000";
  }

  /**
   * Send a user message to the backend. Backend will automatically
   * classify intent as chitchat or product_search.
   * If product_search, response will include search_results.
   */
  async sendMessage(message: string): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseURL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          top_k: 20,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      return data;
    } catch (error) {
      console.error('API Error:', error);
      throw new Error('Failed to connect to ZeeVice AI backend');
    }
  }
}

export const api = new ZeeViceAPI();
